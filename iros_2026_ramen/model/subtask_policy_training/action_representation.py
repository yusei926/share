"""Action representations used by the flip-table ACT and Diffusion baselines.

The source dataset stores executable, absolute G1 joint targets.  The
``chunk_relative_arm_absolute_gripper`` representation is a model-space
transform only: every arm action in a chunk is expressed relative to the
measured arm state at the chunk start.  It is never integrated from a previous
model prediction, so a new observation reanchors the next chunk.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch


ABSOLUTE_TARGET = "absolute_target"
CHUNK_RELATIVE_ARM_ABSOLUTE_GRIPPER = "chunk_relative_arm_absolute_gripper"
SUPPORTED_REPRESENTATIONS = (ABSOLUTE_TARGET, CHUNK_RELATIVE_ARM_ABSOLUTE_GRIPPER)

STATE_DIM = 19
ACTION_DIM = 16
ARM_DIM = 14
STATE_ARM_START = 3
CHUNK_SIZE = 16


def validate_representation(value: str) -> str:
    value = str(value).strip()
    if value not in SUPPORTED_REPRESENTATIONS:
        raise ValueError(
            f"unsupported action representation {value!r}; expected one of {SUPPORTED_REPRESENTATIONS}"
        )
    return value


def semantics(value: str) -> str:
    value = validate_representation(value)
    if value == ABSOLUTE_TARGET:
        return "14 absolute arm joint targets plus 2 absolute Dex1 gripper commands"
    return (
        "14 arm joint targets relative to the measured arm state at chunk start; "
        "2 absolute Dex1 gripper commands; no recursive action integration"
    )


def encode_action_chunk(
    actions: torch.Tensor,
    observation_state: torch.Tensor,
    representation: str,
) -> torch.Tensor:
    """Transform executable action targets into the model's action space.

    ``actions`` is ``[batch, horizon, 16]`` and ``observation_state`` is
    ``[batch, observations, 19]``.  The newest state is the one aligned to the
    first target action.  The input is never modified in place.
    """

    representation = validate_representation(representation)
    if actions.ndim != 3 or actions.shape[-1] != ACTION_DIM:
        raise ValueError(f"actions must be [batch, horizon, {ACTION_DIM}], got {tuple(actions.shape)}")
    if observation_state.ndim != 3 or observation_state.shape[-1] != STATE_DIM:
        raise ValueError(
            f"observation_state must be [batch, observations, {STATE_DIM}], got {tuple(observation_state.shape)}"
        )
    if representation == ABSOLUTE_TARGET:
        return actions
    result = actions.clone()
    reference = observation_state[:, -1, STATE_ARM_START : STATE_ARM_START + ARM_DIM]
    result[:, :, :ARM_DIM] -= reference.unsqueeze(1)
    return result


def decode_action_chunk(
    actions: torch.Tensor,
    observation_state: torch.Tensor,
    representation: str,
) -> torch.Tensor:
    """Convert model-space action chunks back to executable absolute targets."""

    representation = validate_representation(representation)
    if representation == ABSOLUTE_TARGET:
        return actions
    if actions.ndim != 3 or actions.shape[-1] != ACTION_DIM:
        raise ValueError(f"actions must be [batch, horizon, {ACTION_DIM}], got {tuple(actions.shape)}")
    if observation_state.ndim != 3 or observation_state.shape[-1] != STATE_DIM:
        raise ValueError(
            f"observation_state must be [batch, observations, {STATE_DIM}], got {tuple(observation_state.shape)}"
        )
    reference = observation_state[:, -1, STATE_ARM_START : STATE_ARM_START + ARM_DIM]
    result = actions.clone()
    result[:, :, :ARM_DIM] += reference.unsqueeze(1)
    return result


def load_training_stats(dataset_root: Path, representation: str) -> dict[str, Any]:
    """Load train-only state stats and action stats in the requested model space."""

    representation = validate_representation(representation)
    stats = json.loads((dataset_root / "meta" / "stats.json").read_text(encoding="utf-8"))
    if representation == ABSOLUTE_TARGET:
        return stats
    stats["action"] = compute_chunk_relative_action_stats(dataset_root)
    stats["action_representation"] = representation
    return stats


def compute_chunk_relative_action_stats(dataset_root: Path) -> dict[str, Any]:
    """Compute train-only statistics in the exact model action space.

    Each action chunk is anchored once at ``q_current[t]``.  Statistics must
    use that same anchor for every valid target in the chunk; subtracting a
    per-frame state here would silently make training and inference normalize
    different quantities.
    """

    split = json.loads((dataset_root / "meta" / "team_ramen_episode_split.json").read_text(encoding="utf-8"))
    train_episodes = {int(value) for value in split["splits"]["train"]["episode_indices"]}
    if not train_episodes:
        raise ValueError("training split has no episodes")

    sums = np.zeros(ACTION_DIM, dtype=np.float64)
    squared_sums = np.zeros(ACTION_DIM, dtype=np.float64)
    minimum = np.full(ACTION_DIM, np.inf, dtype=np.float64)
    maximum = np.full(ACTION_DIM, -np.inf, dtype=np.float64)
    count = 0
    columns = ["episode_index", "frame_index", "observation.state", "action"]
    by_episode: dict[int, list[dict[str, Any]]] = {}
    for path in sorted((dataset_root / "data").glob("chunk-*/*.parquet")):
        table = pq.read_table(path, columns=columns)
        for row in table.to_pylist():
            episode = int(row["episode_index"])
            if episode in train_episodes:
                by_episode.setdefault(episode, []).append(row)
    for episode_rows in by_episode.values():
        episode_rows.sort(key=lambda row: int(row["frame_index"]))
        states = np.asarray([row["observation.state"] for row in episode_rows], dtype=np.float64)
        targets = np.asarray([row["action"] for row in episode_rows], dtype=np.float64)
        if states.ndim != 2 or states.shape[1] != STATE_DIM:
            raise ValueError(f"invalid state shape {states.shape}")
        if targets.ndim != 2 or targets.shape[1] != ACTION_DIM:
            raise ValueError(f"invalid action shape {targets.shape}")
        if not np.isfinite(states).all() or not np.isfinite(targets).all():
            raise ValueError("chunk-relative action statistics require finite state and action values")
        for chunk_start in range(len(episode_rows)):
            chunk_end = min(chunk_start + CHUNK_SIZE, len(episode_rows))
            values = targets[chunk_start:chunk_end].copy()
            values[:, :ARM_DIM] -= states[
                chunk_start, STATE_ARM_START : STATE_ARM_START + ARM_DIM
            ]
            sums += values.sum(axis=0)
            squared_sums += np.square(values).sum(axis=0)
            minimum = np.minimum(minimum, values.min(axis=0))
            maximum = np.maximum(maximum, values.max(axis=0))
            count += len(values)
    if count == 0 or not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
        raise ValueError("could not compute chunk-relative action statistics")
    mean = sums / count
    variance = np.maximum(squared_sums / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or math.isclose(float(std.max()), 0.0):
        raise ValueError("chunk-relative action statistics are non-finite or degenerate")
    return {
        "count": [count],
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min": minimum.tolist(),
        "max": maximum.tolist(),
    }
