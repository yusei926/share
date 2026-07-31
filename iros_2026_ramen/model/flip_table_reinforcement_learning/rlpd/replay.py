"""Bounded CPU replay storage with balanced prior/online sampling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class ReplayBatch:
    observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_observation: torch.Tensor
    done: torch.Tensor

    def to(self, device: torch.device | str) -> "ReplayBatch":
        return ReplayBatch(
            *(value.to(device, non_blocking=True) for value in self.__dict__.values())
        )

    @staticmethod
    def concatenate(*batches: "ReplayBatch") -> "ReplayBatch":
        if not batches:
            raise ValueError("at least one replay batch is required")
        return ReplayBatch(
            *(torch.cat([getattr(batch, field) for batch in batches], dim=0) for field in ReplayBatch.__dataclass_fields__)
        )


class ReplayBuffer:
    def __init__(self, capacity: int, observation_dim: int, action_dim: int = 16) -> None:
        if capacity <= 0 or observation_dim <= 0 or action_dim <= 0:
            raise ValueError("replay dimensions and capacity must be positive")
        self.capacity = int(capacity)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.observation = np.empty((capacity, observation_dim), dtype=np.float32)
        self.action = np.empty((capacity, action_dim), dtype=np.float32)
        self.reward = np.empty((capacity, 1), dtype=np.float32)
        self.next_observation = np.empty((capacity, observation_dim), dtype=np.float32)
        self.done = np.empty((capacity, 1), dtype=np.float32)
        self.size = 0
        self.position = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        observation: torch.Tensor | np.ndarray,
        action: torch.Tensor | np.ndarray,
        reward: torch.Tensor | np.ndarray,
        next_observation: torch.Tensor | np.ndarray,
        done: torch.Tensor | np.ndarray,
    ) -> None:
        arrays = [
            np.asarray(value.detach().cpu() if torch.is_tensor(value) else value, dtype=np.float32)
            for value in (observation, action, reward, next_observation, done)
        ]
        obs, act, rew, next_obs, terminal = arrays
        if obs.ndim != 2 or obs.shape[1] != self.observation_dim:
            raise ValueError(f"observation must be [B,{self.observation_dim}], got {obs.shape}")
        batch = obs.shape[0]
        expected = (
            (batch, self.action_dim),
            (batch, 1),
            (batch, self.observation_dim),
            (batch, 1),
        )
        for name, value, shape in zip(
            ("action", "reward", "next_observation", "done"),
            (act, rew, next_obs, terminal),
            expected,
        ):
            if value.shape != shape:
                raise ValueError(f"{name} must be {shape}, got {value.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or Inf")
        if not np.isfinite(obs).all():
            raise ValueError("observation contains NaN or Inf")

        if batch >= self.capacity:
            arrays = [value[-self.capacity :] for value in arrays]
            obs, act, rew, next_obs, terminal = arrays
            batch = self.capacity
        indices = (np.arange(batch) + self.position) % self.capacity
        for storage, value in zip(
            (self.observation, self.action, self.reward, self.next_observation, self.done),
            (obs, act, rew, next_obs, terminal),
        ):
            storage[indices] = value
        self.position = int((self.position + batch) % self.capacity)
        self.size = min(self.capacity, self.size + batch)

    def sample(self, batch_size: int, *, rng: np.random.Generator | None = None) -> ReplayBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.size < batch_size:
            raise ValueError(f"replay contains {self.size} rows, fewer than requested {batch_size}")
        generator = rng or np.random.default_rng()
        indices = generator.integers(0, self.size, size=batch_size)
        return ReplayBatch(
            torch.from_numpy(self.observation[indices]),
            torch.from_numpy(self.action[indices]),
            torch.from_numpy(self.reward[indices]),
            torch.from_numpy(self.next_observation[indices]),
            torch.from_numpy(self.done[indices]),
        )

    def save(self, directory: str | Path) -> None:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": "team_ramen_replay_v1",
            "capacity": self.capacity,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "size": self.size,
            "position": self.position,
        }
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        for name in ("observation", "action", "reward", "next_observation", "done"):
            np.save(output / f"{name}.npy", getattr(self, name)[: self.size], allow_pickle=False)

    def restore(self, directory: str | Path) -> None:
        source = Path(directory)
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("schema_version") != "team_ramen_replay_v1":
            raise ValueError("unsupported replay checkpoint schema")
        expected = {
            "capacity": self.capacity,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
        }
        for name, value in expected.items():
            if int(metadata[name]) != value:
                raise ValueError(f"replay {name} mismatch: {metadata[name]} != {value}")
        size = int(metadata["size"])
        position = int(metadata["position"])
        if not 0 <= size <= self.capacity or not 0 <= position < self.capacity:
            raise ValueError("replay size or position is invalid")
        expected_shapes = {
            "observation": (size, self.observation_dim),
            "action": (size, self.action_dim),
            "reward": (size, 1),
            "next_observation": (size, self.observation_dim),
            "done": (size, 1),
        }
        for name, shape in expected_shapes.items():
            value = np.load(source / f"{name}.npy", allow_pickle=False)
            if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"replay {name} has invalid shape, dtype or values")
            getattr(self, name)[:size] = value
        self.size = size
        self.position = position


def balanced_replay_sample(
    prior: ReplayBuffer,
    online: ReplayBuffer,
    batch_size: int,
    *,
    prior_fraction: float = 0.5,
    rng: np.random.Generator | None = None,
) -> ReplayBatch:
    if not 0.0 <= prior_fraction <= 1.0:
        raise ValueError("prior_fraction must be in [0, 1]")
    prior_count = int(round(batch_size * prior_fraction))
    online_count = batch_size - prior_count
    batches = []
    if prior_count:
        batches.append(prior.sample(prior_count, rng=rng))
    if online_count:
        batches.append(online.sample(online_count, rng=rng))
    return ReplayBatch.concatenate(*batches)
